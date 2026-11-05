"""P0 nominal native replay REQUEST recorder (isolated and minimal; lineage r24-resume-p0-nominal-v1, UNQUALIFIED).

Replays one frozen trace_workload schema-1 replay trace against a loopback gateway at ABSOLUTE offered offsets and
records per-request SSE evidence: finish_reason (observed / absent / contradiction), usage, first/last content,
[DONE], HTTP status, fate, request ids and scheduled/sent clocks.  ``normalized_row`` emits the STRICT
``nominal_metrics.ROW_KEYS`` rows with the unchanged r23 natural-SLO fields, so ``build_run`` hands
``nominal_metrics.compute`` a run without another adapter.

Owned by the DRIVER, never by this recorder: model-server / gateway launch, warmup, static budget installation,
final gateway drain, gateway tokenizer and zero-character-fallback checks, full-wall start/end, controller decision
intervals and FIFO/FCFS verification.  No winner, capacity, SLO guarantee or qualification is claimed.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time
import urllib.parse
import uuid

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [path for path in (str(HERE), str(ROOT)) if path not in sys.path]

import nominal_metrics as nm  # noqa: E402
import telemetry  # noqa: E402
import trace_workload as tw  # noqa: E402

SCHEMA = 'r24-resume-p0-nominal-client-v1'
BINDING_SCHEMA = 'r24-resume-p0-nominal-model-binding-v1'
BINDING_KEYS = frozenset({'schema', 'served_alias', 'local_model_path', 'tokenizer_sha256', 'served_model_names',
                          'backend_argv', 'backend_argv_source'})
CHAT_PATH = '/v1/chat/completions'
MODELS_PATH = '/v1/models'
REQUEST_ID_HEADER = 'x-benchmark-request-id'
RESPONSE_ID_HEADER = 'x-request-id'
ADMISSION_DETAIL = 'request rejected: admission capacity exceeded'
IGNORE_EOS = False
LOOPBACK_HOSTS = frozenset({'127.0.0.1', 'localhost', '::1'})
MAX_START_WAIT_S = 600.0
STATUSES = ('collected', 'refused', 'interrupted', 'failed')
RAW_PURPOSE_RULE = ('raw-only engineering purpose: ^[a-z][a-z0-9_]*$ without "eval" or "held"; equals the trace '
                    'provenance purpose; never finalized into a core train/eval metrics run')
CONTROL_EVIDENCE_KEYS = frozenset({'acknowledged_at', 'held_verified_at', 'clock_domain', 'source'})
# Precedence follows sustained_load.consume_request (InvalidSSE > IncompleteStream > EmptyContentStream).
ANOMALIES = ('invalid_sse', 'error_event', 'unexpected_choice_index', 'finish_reason_contradiction',
             'data_after_done', 'missing_done', 'empty_content')
FINISH_EVIDENCE = ('observed', 'absent', 'contradiction', 'not_applicable')
TAPE_KEYS = ('kind', 'offered_tape_schema', 'offered_tape_sha256', 'generator', 'protocol_id', 'purpose', 'source_id',
             'tokenizer_sha256', 'measured_equals_requested_count')
SOURCES = {'nominal_client.py': Path(__file__).resolve(), 'nominal_metrics.py': HERE / 'nominal_metrics.py',
           'telemetry.py': ROOT / 'telemetry.py', 'trace_workload.py': ROOT / 'trace_workload.py',
           'score_http.py': nm.SCORE_HTTP_PATH}
ROW_MAPPING = {
    'request_id': 'trace_id', 'group_tokens': 'trace input_tokens (tokenizer-measured, frozen before outcomes)',
    'max_output_tokens': 'trace max_tokens', 'scheduled_at': 'offered_start + arrival_seconds',
    'sent_at': 'time.monotonic() immediately before client.stream (send-attempt start, before connection '
               'acquisition; null if never sent)',
    'first_content_at': 'time.monotonic() when the first content-bearing SSE event was parsed',
    'terminal_at': 'time.monotonic() in finally after the response context closed (null if never sent)',
    'completion_tokens': 'the single usage.completion_tokens object of the stream, else null',
    'slo.status': 'fate', 'slo.finish_reason': 'observed finish_reason (null when absent, contradictory or n/a)',
    'slo.requested_ignore_eos': 'ignore_eos of the payload actually sent (constant false)',
    'slo.stream_done': 'data: [DONE] received', 'slo.stream_error': 'fate == stream_error',
}
QUALIFIERS = [
    'UNQUALIFIED nominal lineage: no G06/G16/Native14 qualification, winner, capacity or SLO guarantee',
    'clocks are client HTTP observations (send-attempt start, SSE parse times, close), not per-token GPU clocks',
    'model binding verified here: alias == trace.model, backend argv model path and served names, backend '
    '/v1/models ids/root/max_model_len, tokenizer fingerprint and every prompt token count from local_model_path; '
    'NOT verified here: gateway MODEL_NAME, gateway tokenizer_available, zero character fallback, CostSpec identity',
    'FIFO label pending: an argv --scheduling-policy value is recorded, never claimed as verified FCFS ordering',
    'no warmup, budget installation, permit, server launch or gateway drain by this recorder',
    'distinct hashes are integrity pins, not Train/Eval disjointness or leakage proof',
    'purposes other than train/eval are raw engineering captures only (never core metrics runs, never heldout)',
]
NOT_CLIENT_OWNED = [
    'backend launch and actual argv capture (binding.backend_argv/backend_argv_source)',
    'gateway launch with MODEL_NAME == local_model_path; tokenizer_available; zero character fallback; CostSpec',
    'warmup and post-warmup gateway drain before the offered start',
    'static fixed budget as startup MAX_ADMITTED_COST before warmup/offers: acknowledged before offered_start and '
    'verified held through final drain (control + control_evidence)',
    'one shared absolute host-monotonic offered START with lead time for the recorder preflight (binding, tokenizer '
    'load, prompt verification, /v1/models); a passed START is refused with zero sends, never retimed',
    'final gateway drain (in_flight 0, admitted cost 0) and fallback counters after the recorder returns',
    'full wall record {start, end, clock_domain, source} covering setup, solver, HTTP, unresolved and final drain',
    'controller decision intervals (coordinator journal) for non-fixed arms',
    'FIFO/FCFS backend ordering verification',
]


class ClientRefusal(ValueError):
    """A prerequisite or input the recorder cannot honour truthfully: refused, never repaired."""


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _reject_constant(token):
    raise ValueError(f'non-finite JSON constant {token}')


def _strict_json(data, label):
    """JSON without duplicate keys or NaN/Infinity, so every capture stays writable with allow_nan=False."""
    def pairs(items):
        if len({key for key, _ in items}) != len(items):
            raise ValueError(f'{label}: duplicate JSON object key')
        return dict(items)

    return json.loads(data, object_pairs_hook=pairs, parse_constant=_reject_constant)


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError('must be a finite number')
    return value


def _purpose(text):
    if text in nm.PURPOSES:
        return text
    if (not text or not text[0].islower() or not text.isascii()
            or any(not (char.islower() or char.isdigit() or char == '_') for char in text)
            or 'eval' in text or 'held' in text):
        raise argparse.ArgumentTypeError(f'core {nm.PURPOSES} or {RAW_PURPOSE_RULE}')
    return text


def _sha_hex(value):
    return isinstance(value, str) and len(value) == 64 and all(char in '0123456789abcdef' for char in value)


def _option_values(argv, option):
    """Values of a space-separated option; ``--opt=value`` or repeated forms return None (refused by callers)."""
    spelled = [index for index, arg in enumerate(argv) if arg.replace('_', '-').split('=', 1)[0] == option]
    if not spelled:
        return []
    if len(spelled) != 1 or argv[spelled[0]] != option:
        return None
    values = []
    for arg in argv[spelled[0] + 1:]:
        if arg.startswith('-'):
            break
        values.append(arg)
    return values


def check_binding(trace, binding):
    """Pure strict model binding check (no I/O); returns argv evidence or raises ClientRefusal."""
    if binding is None:
        raise ClientRefusal('unbound served alias: an explicit model binding record is required')
    if not isinstance(binding, dict) or set(binding) != BINDING_KEYS or binding['schema'] != BINDING_SCHEMA:
        raise ClientRefusal(f'binding keys must be exactly {sorted(BINDING_KEYS)} with schema {BINDING_SCHEMA}')
    alias, path, names = binding['served_alias'], binding['local_model_path'], binding['served_model_names']
    if not isinstance(alias, str) or not alias or alias != trace['model']:
        raise ClientRefusal('served_alias must equal the frozen trace.model (trace.model is never changed)')
    if not isinstance(path, str) or not path.startswith('/') or os.path.normpath(path) != path:
        raise ClientRefusal('local_model_path must be an absolute normalized path')
    provenance = trace.get('provenance')
    expected = provenance.get('tokenizer_sha256') if isinstance(provenance, dict) else None
    if not _sha_hex(binding['tokenizer_sha256']) or binding['tokenizer_sha256'] != expected:
        raise ClientRefusal('tokenizer_sha256 must equal the trace provenance tokenizer_sha256')
    if (not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or alias not in names):
        raise ClientRefusal('served_model_names must be unique non-empty strings including served_alias')
    argv, source = binding['backend_argv'], binding['backend_argv_source']
    if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) for arg in argv):
        raise ClientRefusal('backend_argv must be the captured non-empty argv list')
    if not isinstance(source, str) or not source:
        raise ClientRefusal('backend_argv_source must state how the driver captured the actual argv')
    launches = any(argv[index:index + 2] in (['serve', path], ['--model', path]) for index in range(len(argv))) \
        or f'--model={path}' in argv
    if not launches:
        raise ClientRefusal('backend_argv does not launch local_model_path (serve <path> or --model <path>)')
    if _option_values(argv, '--served-model-name') != names:
        raise ClientRefusal('backend_argv needs exactly one space-separated --served-model-name whose values equal '
                            'served_model_names')
    def recorded(option):
        values = _option_values(argv, option)
        return 'unparsed_form' if values is None else values[0] if len(values) == 1 else values or None

    return {
        'served_alias': alias, 'local_model_path': path, 'served_model_names': list(names),
        'argv_scheduling_policy': recorded('--scheduling-policy'),
        'argv_scheduling_policy_note': 'recorded only; FIFO label stays pending (not a verified FCFS ordering)',
        'argv_prefix_caching_flags': [arg for arg in argv if 'prefix-caching' in arg.replace('_', '-')],
        'argv_max_model_len': recorded('--max-model-len'),
        'backend_argv_sha256': hashlib.sha256(json.dumps(argv).encode()).hexdigest(),
        'backend_argv_source': source,
    }


def check_served_models(binding, models, max_model_len):
    """The backend must serve exactly the bound names; the alias root must be the local path."""
    data = models.get('data') if isinstance(models, dict) else None
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise ClientRefusal('backend /v1/models response is not a model list')
    ids = [item.get('id') for item in data]
    if any(not isinstance(item, str) for item in ids) or len(set(ids)) != len(ids) \
            or sorted(ids) != sorted(binding['served_model_names']):
        raise ClientRefusal('backend /v1/models ids differ from the bound served_model_names')
    entry = next(item for item in data if item['id'] == binding['served_alias'])
    if entry.get('root') != binding['local_model_path']:
        raise ClientRefusal('backend served alias root differs from local_model_path')
    if entry.get('max_model_len') != max_model_len:
        raise ClientRefusal('backend served alias max_model_len differs from the trace max_model_len')
    return {'ids': ids, 'alias_root': entry['root'], 'alias_max_model_len': entry['max_model_len']}


def check_tokenizer(trace, binding, tokenizer):
    """The tokenizer loaded from local_model_path must reproduce the fingerprint and every trace prompt count."""
    fingerprint = tw.tokenizer_fingerprint(tokenizer)
    if fingerprint != binding['tokenizer_sha256']:
        raise ClientRefusal('tokenizer loaded from local_model_path does not match the bound fingerprint')
    try:
        tw.verify_prompts(trace, tokenizer)
    except ValueError as exc:
        raise ClientRefusal(f'trace prompt token counts differ under the bound tokenizer: {exc}') from exc
    return {'fingerprint': fingerprint, 'prompts_verified': len(trace['requests'])}


def load_tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


class StreamParse:
    """Pure SSE evidence accumulator: ``feed(line, at)`` per received line with the caller's monotonic time."""

    def __init__(self):
        self.data_events = self.content_events = self.finish_events = 0
        self.first_content_at = self.last_content_at = self.done_at = None
        self.finish_reasons, self.usages, self.anomalies = [], [], []

    def _flag(self, name):
        if name not in self.anomalies:
            self.anomalies.append(name)

    def feed(self, line, at):
        if not line.startswith('data:'):
            return
        data = line[5:].strip()
        if self.done_at is not None:
            self._flag('data_after_done')
            return
        if data == '[DONE]':
            self.done_at = at
            return
        self.data_events += 1
        try:
            event = _strict_json(data, 'SSE data')
        except ValueError:
            self._flag('invalid_sse')
            return
        try:
            content = telemetry.has_sse_content(event)
        except ValueError:
            self._flag('error_event' if isinstance(event, dict) and event.get('error') is not None else 'invalid_sse')
            return
        if event.get('usage') is not None:
            self.usages.append(event['usage'])
        finished_before = bool(self.finish_reasons)
        for choice in event.get('choices', []):
            index = choice.get('index', 0)
            if type(index) is not int or index != 0:
                self._flag('unexpected_choice_index')
            reason = choice.get('finish_reason')
            if reason is None:
                continue
            if not isinstance(reason, str):
                self._flag('invalid_sse')
                continue
            self.finish_events += 1
            if reason not in self.finish_reasons:
                self.finish_reasons.append(reason)
        if content:
            if finished_before:
                self._flag('finish_reason_contradiction')
            self.content_events += 1
            if self.first_content_at is None:
                self.first_content_at = at
            self.last_content_at = at

    def result(self):
        anomalies = set(self.anomalies)
        if len(self.finish_reasons) > 1:
            anomalies.add('finish_reason_contradiction')
        if self.done_at is None:
            anomalies.add('missing_done')
        if not self.content_events:
            anomalies.add('empty_content')
        if 'finish_reason_contradiction' in anomalies:
            evidence, finish = 'contradiction', None
        elif self.finish_reasons:
            evidence, finish = 'observed', self.finish_reasons[0]
        else:
            evidence, finish = 'absent', None
        usage, tokens = (self.usages[0], None) if len(self.usages) == 1 else (None, None)
        if not self.usages:
            tokens_evidence = 'absent'
        elif len(self.usages) > 1:
            tokens_evidence = 'multiple_usage'
        else:
            value = usage.get('completion_tokens')
            tokens, tokens_evidence = (value, 'usage') if type(value) is int and value >= 0 else (None, 'invalid_usage')
        prompt = usage.get('prompt_tokens') if usage is not None else None
        return {
            'first_content_at': self.first_content_at, 'last_content_at': self.last_content_at,
            'done_at': self.done_at, 'stream_done': self.done_at is not None, 'data_events': self.data_events,
            'content_events': self.content_events, 'finish_reason': finish, 'finish_reason_evidence': evidence,
            'finish_reasons_seen': list(self.finish_reasons), 'finish_reason_events': self.finish_events,
            'anomalies': [name for name in ANOMALIES if name in anomalies], 'usage': usage,
            'usage_events': len(self.usages), 'completion_tokens': tokens,
            'completion_tokens_evidence': tokens_evidence, 'prompt_tokens': prompt if type(prompt) is int else None,
        }


def prepare_records(trace, offered_start, run_id):
    """One placeholder per offered request (fate not_launched until sent), allocated before any send."""
    namespace = uuid.UUID(run_id)
    return [{
        'trace_id': request['trace_id'], 'client_request_id': uuid.uuid5(namespace, request['trace_id']).hex,
        'request_id': None, 'expected_input_tokens': request['input_tokens'], 'max_tokens': request['max_tokens'],
        'scheduled_at': offered_start + request['arrival_seconds'], 'sent_at': None, 'send_lag_s': None,
        'response_headers_at': None, 'first_content_at': None, 'last_content_at': None, 'done_at': None,
        'terminal_at': None, 'http_status': None, 'fate': 'not_launched', 'fate_reason': 'not_sent',
        'stream_done': False, 'data_events': 0, 'content_events': 0, 'finish_reason': None,
        'finish_reason_evidence': 'not_applicable', 'finish_reasons_seen': [], 'finish_reason_events': 0,
        'anomalies': [], 'usage': None, 'usage_events': 0, 'completion_tokens': None,
        'completion_tokens_evidence': 'not_applicable', 'prompt_tokens': None, 'prompt_tokens_match': None,
        'rejection': None, 'body_bytes': None, 'body_sha256': None, 'error_type': None, 'error_message': None,
    } for request in trace['requests']]


def _apply(record, parsed):
    record.update(parsed)
    if parsed['prompt_tokens'] is not None:
        record['prompt_tokens_match'] = parsed['prompt_tokens'] == record['expected_input_tokens']


def _admission_rejection(status, body):
    if status != 503:
        return None
    try:
        value = _strict_json(body, 'admission body')
    except ValueError:
        return None
    if isinstance(value, dict) and value.get('detail') == ADMISSION_DETAIL and 'estimated_request_cost' in value:
        cost = value['estimated_request_cost']
        return {'detail': ADMISSION_DETAIL, 'estimated_request_cost': cost if _finite(cost) else None}
    return None


def _error(record, exc, late):
    record['error_type'] = type(exc).__name__
    record['error_message'] = ('after_outcome: ' if late else '') + str(exc)[:300]


async def _request(client, url, model, request, record):
    """Send one offered request at its absolute time and record every observation; never reschedules."""
    try:
        payload = {'model': model, 'messages': [{'role': 'user', 'content': request['prompt']}], 'temperature': 0,
                   'max_tokens': request['max_tokens'], 'stream': True, 'stream_options': {'include_usage': True},
                   'ignore_eos': IGNORE_EOS}
        headers = {REQUEST_ID_HEADER: record['client_request_id']}
        while (now := time.monotonic()) < record['scheduled_at']:
            await asyncio.sleep(record['scheduled_at'] - now)
    except asyncio.CancelledError:
        record['fate_reason'] = 'cancelled_before_send'
        raise
    except Exception as exc:
        record['fate_reason'] = 'implementation_exception_before_send'
        _error(record, exc, False)
        return
    record['sent_at'] = time.monotonic()
    record['send_lag_s'] = record['sent_at'] - record['scheduled_at']
    record['fate'], record['fate_reason'] = 'unknown', 'sent_outcome_not_recorded'
    parse = None
    try:
        async with client.stream('POST', url, json=payload, headers=headers) as response:
            record['response_headers_at'] = time.monotonic()
            record['http_status'] = response.status_code
            record['request_id'] = response.headers.get(RESPONSE_ID_HEADER)
            if not response.is_success:
                body = await response.aread()
                record['body_bytes'], record['body_sha256'] = len(body), hashlib.sha256(body).hexdigest()
                rejection = _admission_rejection(response.status_code, body)
                record['rejection'] = rejection
                record['fate'], record['fate_reason'] = (('rejected', 'admission_capacity_503') if rejection else
                                                         ('http_error', f'http_{response.status_code}'))
            else:
                parse = StreamParse()
                async for line in response.aiter_lines():
                    parse.feed(line, time.monotonic())
                parsed = parse.result()
                _apply(record, parsed)
                record['fate'], record['fate_reason'] = (('stream_error', parsed['anomalies'][0])
                                                         if parsed['anomalies'] else ('complete', 'complete'))
    except asyncio.CancelledError as exc:
        late = record['fate'] != 'unknown'
        if not late:
            if parse is not None:
                _apply(record, parse.result())
            record['fate'], record['fate_reason'] = 'cancelled', ('before_response_headers'
                                                                  if record['http_status'] is None else 'response_body')
        _error(record, exc, late)
        raise
    except httpx.HTTPError as exc:
        late = record['fate'] != 'unknown'
        if not late:
            if parse is not None:
                _apply(record, parse.result())
            record['fate'], record['fate_reason'] = 'transport_error', ('before_response_headers'
                                                                        if record['http_status'] is None
                                                                        else 'response_body')
        _error(record, exc, late)
    except Exception as exc:
        late = record['fate'] != 'unknown'
        if not late:
            record['fate_reason'] = 'implementation_exception'
        _error(record, exc, late)
    finally:
        record['terminal_at'] = time.monotonic()


async def replay(client, trace, records, *, gateway_url, offered_start, drain_timeout):
    """Offer every request at offered_start + arrival_seconds; retain every outcome in ``records`` (in place).

    In-flight requests are cancelled at offered_end + drain_timeout.  Returns the drain deadline.
    """
    if not _finite(drain_timeout) or drain_timeout < 0:
        raise ClientRefusal('drain_timeout must be a finite non-negative number')
    if len(records) != len(trace['requests']):
        raise ClientRefusal('records must cover the offered trace one-to-one')
    if not time.monotonic() < offered_start:
        raise ClientRefusal('offered_start already passed: absolute offsets cannot be honoured (no rescheduling)')
    url = gateway_url.rstrip('/') + CHAT_PATH
    tasks = [asyncio.create_task(_request(client, url, trace['model'], request, record))
             for request, record in zip(trace['requests'], records)]
    offered_end = offered_start + trace['duration_seconds']
    deadline = offered_end + drain_timeout
    try:
        while (now := time.monotonic()) < offered_end:
            await asyncio.sleep(offered_end - now)
        pending = [task for task in tasks if not task.done()]
        if pending:
            await asyncio.wait(pending, timeout=max(0.0, deadline - time.monotonic()))
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return deadline


def normalized_row(record):
    """The strict nominal_metrics row of one per-request evidence record (see ROW_MAPPING)."""
    fate = record['fate']
    return {
        'request_id': record['trace_id'], 'group_tokens': record['expected_input_tokens'],
        'max_output_tokens': record['max_tokens'], 'scheduled_at': record['scheduled_at'],
        'sent_at': record['sent_at'], 'first_content_at': record['first_content_at'],
        'terminal_at': record['terminal_at'], 'fate': fate, 'completion_tokens': record['completion_tokens'],
        'slo': {'status': fate, 'finish_reason': record['finish_reason'], 'requested_ignore_eos': IGNORE_EOS,
                'stream_done': record['stream_done'], 'stream_error': fate == 'stream_error'},
    }


def summarize_records(records):
    def counts(key, values):
        out = {value: 0 for value in values}
        for record in records:
            out[str(record[key])] = out.get(str(record[key]), 0) + 1
        return out

    lags = [record['send_lag_s'] for record in records if record['send_lag_s'] is not None]
    finish = {}
    for record in records:
        if record['fate'] == 'complete':
            finish[str(record['finish_reason'])] = finish.get(str(record['finish_reason']), 0) + 1
    return {
        'offered': len(records), 'sent': len(lags), 'fates': counts('fate', nm.FATES),
        'finish_reason_evidence': counts('finish_reason_evidence', FINISH_EVIDENCE),
        'finish_reasons_of_complete': finish,
        'completion_tokens_evidence': counts('completion_tokens_evidence', ()),
        'prompt_tokens_mismatch': sum(record['prompt_tokens_match'] is False for record in records),
        'prompt_tokens_matched': sum(record['prompt_tokens_match'] is True for record in records),
        'implementation_exceptions': sum(record['fate_reason'] in ('implementation_exception',
                                                                   'implementation_exception_before_send')
                                         for record in records),
        'max_send_lag_s': max(lags) if lags else None,
    }


def clock_domain():
    """Host-wide CLOCK_MONOTONIC identity; a driver record must carry the identical string."""
    try:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:
        boot = 'unavailable'
    info = time.get_clock_info('monotonic')
    return f'time.monotonic {info.implementation} host={socket.gethostname()} boot_id={boot}'


def build_run(capture, *, control=None, control_missing=None, control_evidence=None, full_wall=None,
              capture_sha256=None):
    """nominal_metrics run for a collected capture; full wall and control facts come only from driver records."""
    if not isinstance(capture, dict) or capture.get('schema') != SCHEMA:
        raise ClientRefusal(f'not a {SCHEMA} capture')
    if capture['status'] != 'collected':
        raise ClientRefusal(f'capture status {capture["status"]!r}: preserved as failure evidence, not scored')
    if capture['purpose'] not in nm.PURPOSES:
        raise ClientRefusal(f'purpose {capture["purpose"]!r} is a raw engineering capture, not a core train/eval '
                            'metrics run')
    rows = [normalized_row(record) for record in capture['records']]
    if rows != capture['rows']:
        raise ClientRefusal('capture rows differ from the per-request evidence mapping')
    domain, offered_start = capture['clock_domain'], capture['offered_start']
    observed_end = max([offered_start + capture['offered_duration_s']]
                       + [row['terminal_at'] for row in rows if row['terminal_at'] is not None])
    collected_end = max(observed_end, capture['collection_end'])
    if full_wall is None:
        scope, start, end = 'offered_cohort_observed', offered_start, observed_end
    else:
        if (not isinstance(full_wall, dict) or set(full_wall) != {'start', 'end', 'clock_domain', 'source'}
                or full_wall['clock_domain'] != domain or not _finite(full_wall['start'])
                or not _finite(full_wall['end']) or not isinstance(full_wall['source'], str)
                or not full_wall['source']):
            raise ClientRefusal('full_wall must be a driver record {start, end, clock_domain, source} in the capture '
                                'clock domain')
        if full_wall['start'] > offered_start or full_wall['end'] < collected_end:
            raise ClientRefusal('full_wall must contain the offered window, every terminal and the collection end')
        scope, start, end = 'full_wall', full_wall['start'], full_wall['end']
    if isinstance(control, dict) and control.get('kind') == 'static_fixed':
        evidence = control_evidence
        if (not isinstance(evidence, dict) or set(evidence) != CONTROL_EVIDENCE_KEYS
                or evidence['clock_domain'] != domain or not _finite(evidence['acknowledged_at'])
                or not _finite(evidence['held_verified_at']) or not evidence['acknowledged_at'] < offered_start
                or evidence['held_verified_at'] < collected_end or not isinstance(evidence['source'], str)
                or not evidence['source']):
            raise ClientRefusal('static_fixed control needs driver control_evidence {acknowledged_at < offered_start, '
                                'held_verified_at >= collection end, clock_domain, source}')
    elif control_evidence is not None:
        raise ClientRefusal('control_evidence applies only to a static_fixed control')
    run = {
        'arm': capture['arm'], 'purpose': capture['purpose'], 'clock_domain': domain, 'offered_start': offered_start,
        'offered_duration_s': capture['offered_duration_s'], 'envelope_start': start, 'envelope_end': end,
        'envelope_scope': scope, 'rows': rows, 'slo_missing': None, 'control': control,
        'control_missing': control_missing,
        'provenance': {
            'source': 'nominal_client.py capture', 'capture_sha256': capture_sha256,
            'offered_trace_sha256': capture['trace_sha256'], 'offered_tape': capture['trace_provenance'],
            'run_id': capture['run_id'], 'model_binding': capture['binding_evidence'],
            'client_sources_sha256': capture['sources_sha256'], 'capture_summary': capture['summary'],
            'control_evidence': control_evidence, 'full_wall': full_wall, 'mapping': ROW_MAPPING,
            'qualifiers': list(QUALIFIERS), 'not_client_owned': list(NOT_CLIENT_OWNED),
        },
    }
    try:
        nm.validate_run(run)
    except nm.MetricsInputError as exc:
        raise ClientRefusal(f'normalized run invalid: {exc}') from exc
    return run


def _strict_load(path):
    """(value, bytes) of a strict JSON file; unreadable or non-strict input is a ClientRefusal."""
    try:
        data = Path(path).read_bytes()
        return _strict_json(data, str(path)), data
    except (OSError, ValueError) as exc:
        raise ClientRefusal(f'{path}: unreadable or non-strict JSON: {type(exc).__name__}: {exc}') from exc


def _write_new(path, value):
    data = (json.dumps(value, sort_keys=True, indent=1, allow_nan=False) + '\n').encode()
    with open(path, 'xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(data).hexdigest()


def _loopback(url, name):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'http' or parsed.hostname not in LOOPBACK_HOSTS or parsed.path not in ('', '/'):
        raise ClientRefusal(f'{name} must be an http loopback base URL without a path')
    return url.rstrip('/')


async def _capture(args, capture, trace, binding, transport, tokenizer_loader):
    capture['binding_evidence'] = check_binding(trace, binding)
    try:
        tokenizer = tokenizer_loader(binding['local_model_path'])
    except Exception as exc:
        raise ClientRefusal(f'tokenizer not loadable from local_model_path: {type(exc).__name__}: {exc}') from exc
    capture['binding_evidence']['tokenizer'] = check_tokenizer(trace, binding, tokenizer)
    limits = httpx.Limits(max_connections=len(trace['requests']) + 1,
                          max_keepalive_connections=len(trace['requests']) + 1)
    async with httpx.AsyncClient(transport=transport, timeout=args.request_timeout, limits=limits,
                                 trust_env=False) as client:
        try:
            response = await client.get(capture['config']['backend_url'] + MODELS_PATH)
            response.raise_for_status()
            capture['models_response'] = _strict_json(response.content, 'backend /v1/models')
        except (httpx.HTTPError, ValueError) as exc:
            raise ClientRefusal(f'backend /v1/models unavailable: {type(exc).__name__}: {exc}') from exc
        capture['binding_evidence']['served_models'] = check_served_models(binding, capture['models_response'],
                                                                           trace['max_model_len'])
        now = time.monotonic()
        offered_start = args.offered_start if args.offered_start is not None else now + args.start_delay
        if not now < offered_start <= now + MAX_START_WAIT_S:
            raise ClientRefusal(f'offered_start must lie in (now, now + {MAX_START_WAIT_S}s]: no rescheduling')
        capture['offered_start'] = offered_start
        capture['offered_end'] = offered_start + trace['duration_seconds']
        records = capture['records'] = prepare_records(trace, offered_start, capture['run_id'])
        try:
            capture['drain_deadline'] = offered_start + trace['duration_seconds'] + args.drain_timeout
            await replay(client, trace, records, gateway_url=capture['config']['gateway_url'],
                         offered_start=offered_start, drain_timeout=args.drain_timeout)
        finally:
            capture['collection_end'] = time.monotonic()


def _replay_main(args, argv, transport, tokenizer_loader):
    try:
        args.output_dir.mkdir(mode=0o700)
    except OSError as exc:
        print(json.dumps({'status': 'REFUSED', 'error': f'output dir: {type(exc).__name__}: {exc}'}, sort_keys=True),
              file=sys.stderr)
        return 2
    capture = {
        'schema': SCHEMA, 'status': 'failed', 'error': None, 'arm': args.arm, 'purpose': args.purpose,
        'run_id': str(uuid.uuid4()), 'clock_domain': clock_domain(), 'program': sys.argv[0], 'argv': argv,
        'config': {'gateway_url': None, 'backend_url': None, 'drain_timeout_s': args.drain_timeout,
                   'request_timeout_s': args.request_timeout, 'start_delay_s': args.start_delay,
                   'driver_offered_start': args.offered_start, 'trace_path': str(args.trace),
                   'binding_path': str(args.binding)},
        'host': socket.gethostname(), 'pid': os.getpid(), 'python': sys.version, 'httpx': httpx.__version__,
        'sources_sha256': None,
        'payload_template': {'model': None, 'messages': [{'role': 'user', 'content': '<trace prompt>'}],
                             'temperature': 0, 'max_tokens': '<trace max_tokens>', 'stream': True,
                             'stream_options': {'include_usage': True}, 'ignore_eos': IGNORE_EOS},
        'trace_sha256': None, 'trace_provenance': None, 'binding': None, 'binding_sha256': None,
        'binding_evidence': None, 'models_response': None, 'offered_count': None, 'offered_duration_s': None,
        'offered_start': None, 'offered_end': None, 'drain_deadline': None, 'collection_end': None,
        'records': None, 'rows': None, 'summary': None, 'mapping': ROW_MAPPING, 'qualifiers': QUALIFIERS,
        'not_client_owned': NOT_CLIENT_OWNED,
    }
    code = 4
    try:
        capture['sources_sha256'] = {name: tw.sha256_file(path) for name, path in SOURCES.items()}
        loaded, data = _strict_load(args.trace)
        try:
            trace = tw.validate_trace(loaded)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ClientRefusal(f'trace refused: {type(exc).__name__}: {exc}') from exc
        capture['trace_sha256'] = hashlib.sha256(data).hexdigest()
        provenance = trace.get('provenance') if isinstance(trace.get('provenance'), dict) else {}
        capture['trace_provenance'] = {key: provenance.get(key) for key in TAPE_KEYS}
        capture['payload_template']['model'] = trace['model']
        capture['offered_count'], capture['offered_duration_s'] = len(trace['requests']), trace['duration_seconds']
        capture['binding'], binding_data = _strict_load(args.binding)
        capture['binding_sha256'] = hashlib.sha256(binding_data).hexdigest()
        capture['config']['gateway_url'] = _loopback(args.gateway_url, '--gateway-url')
        capture['config']['backend_url'] = _loopback(args.backend_url, '--backend-url')
        if provenance.get('purpose') != args.purpose:
            raise ClientRefusal(f'trace provenance purpose {provenance.get("purpose")!r} != declared {args.purpose!r}')
        if not _finite(args.drain_timeout) or args.drain_timeout < 0 or not _finite(args.request_timeout) \
                or args.request_timeout <= 0 or not _finite(args.start_delay) or args.start_delay <= 0:
            raise ClientRefusal('timeouts and start delay must be finite (drain >= 0, others > 0)')
        asyncio.run(_run_cancellable(_capture(args, capture, trace, capture['binding'], transport,
                                              tokenizer_loader)))
        capture['status'], code = 'collected', 0
    except ClientRefusal as exc:
        capture['status'], capture['error'], code = 'refused', f'ClientRefusal: {exc}', 2
    except (asyncio.CancelledError, KeyboardInterrupt) as exc:
        capture['status'], capture['error'], code = 'interrupted', f'{type(exc).__name__}: replay interrupted', 3
    except Exception as exc:
        capture['status'], capture['error'] = 'failed', f'{type(exc).__name__}: {str(exc)[:300]}'
    if capture['records'] is not None:
        capture['rows'] = [normalized_row(record) for record in capture['records']]
        capture['summary'] = summarize_records(capture['records'])
    digest = _write_new(args.output_dir / 'capture.json', capture)
    print(json.dumps({'status': capture['status'], 'error': capture['error'], 'capture_sha256': digest,
                      'summary': capture['summary']}, sort_keys=True))
    return code


async def _run_cancellable(coroutine):
    task = asyncio.ensure_future(coroutine)
    loop = asyncio.get_running_loop()
    installed = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous = signal.getsignal(signum)
            loop.add_signal_handler(signum, task.cancel)
            installed.append((signum, previous))
        except (NotImplementedError, RuntimeError, ValueError):
            pass
    try:
        return await task
    finally:
        for signum, previous in installed:
            loop.remove_signal_handler(signum)
            if previous is not None:
                signal.signal(signum, previous)


def _finalize_main(args):
    capture, capture_data = _strict_load(args.capture)
    control = _strict_load(args.control_json)[0] if args.control_json else None
    evidence = _strict_load(args.control_evidence_json)[0] if args.control_evidence_json else None
    full_wall = _strict_load(args.full_wall_json)[0] if args.full_wall_json else None
    run = build_run(capture, control=control, control_missing=args.control_missing, control_evidence=evidence,
                    full_wall=full_wall, capture_sha256=hashlib.sha256(capture_data).hexdigest())
    metrics = nm.compute(run)
    args.output_dir.mkdir(mode=0o700)
    digests = {'run.json': _write_new(args.output_dir / 'run.json', run),
               'metrics.json': _write_new(args.output_dir / 'metrics.json', metrics)}
    print(json.dumps({'status': 'FINALIZED', 'sha256': digests, 'summary': nm.summarize(metrics)}, sort_keys=True))
    return 0


def main(argv=None, *, transport=None, tokenizer_loader=load_tokenizer):
    parser = argparse.ArgumentParser(description='P0 nominal native replay request recorder (no winner, no '
                                                 'qualification).', allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    replay_parser = commands.add_parser('replay', allow_abbrev=False)
    replay_parser.add_argument('--trace', type=Path, required=True)
    replay_parser.add_argument('--binding', type=Path, required=True)
    replay_parser.add_argument('--gateway-url', required=True)
    replay_parser.add_argument('--backend-url', required=True)
    replay_parser.add_argument('--arm', required=True, choices=nm.RUN_ARMS)
    replay_parser.add_argument('--purpose', required=True, type=_purpose)
    replay_parser.add_argument('--output-dir', type=Path, required=True)
    replay_parser.add_argument('--offered-start', type=_finite_float, help='driver absolute time.monotonic() origin')
    replay_parser.add_argument('--start-delay', type=_finite_float, default=0.2)
    replay_parser.add_argument('--drain-timeout', type=_finite_float, default=120.0)
    replay_parser.add_argument('--request-timeout', type=_finite_float, default=120.0)
    finalize_parser = commands.add_parser('finalize', allow_abbrev=False)
    finalize_parser.add_argument('--capture', type=Path, required=True)
    finalize_parser.add_argument('--output-dir', type=Path, required=True)
    finalize_parser.add_argument('--control-json', type=Path)
    finalize_parser.add_argument('--control-evidence-json', type=Path)
    finalize_parser.add_argument('--control-missing')
    finalize_parser.add_argument('--full-wall-json', type=Path)
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    if args.command == 'replay':
        return _replay_main(args, argv, transport, tokenizer_loader)
    try:
        return _finalize_main(args)
    except (ClientRefusal, nm.MetricsInputError, OSError, ValueError) as exc:
        print(json.dumps({'status': 'REFUSED', 'error': f'{type(exc).__name__}: {exc}'}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
