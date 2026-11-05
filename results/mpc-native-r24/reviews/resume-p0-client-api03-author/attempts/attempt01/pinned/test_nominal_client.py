"""Focused CPU controls for nominal_client: literal SSE edge cases, retained fates, strict model binding, CLI.

All HTTP is in-process ``httpx.MockTransport``; no server, network, GPU or real tokenizer is used.
"""
import asyncio
import hashlib
import json
import os
import signal
import time

import httpx
import pytest

import nominal_client as nc
import nominal_metrics as nm
import trace_workload as tw

ALIAS = 'Qwen/Qwen3-4B-Instruct-2507'
LOCAL = '/workspace/pr32-r23/models/Qwen3-4B-Instruct-2507'
GATEWAY = 'http://127.0.0.1:8080'
BACKEND = 'http://127.0.0.1:8000'
RUN_ID = '00000000-0000-4000-8000-000000000001'
DONE = 'data: [DONE]'
USAGE = {'prompt_tokens': 9, 'completion_tokens': 2, 'total_tokens': 11}
MODELS = {'object': 'list', 'data': [{'id': ALIAS, 'object': 'model', 'root': LOCAL, 'max_model_len': 8192}]}


class FakeBackend:
    def to_str(self):
        return '{"literal": "cpu tokenizer"}'


class FakeTokenizer:
    backend_tokenizer = FakeBackend()
    chat_template = 'literal-template'

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
        assert tokenize and add_generation_prompt and return_dict
        return {'input_ids': [0] * (len(messages[0]['content']) + 4)}


FINGERPRINT = tw.tokenizer_fingerprint(FakeTokenizer())


def make_trace(prompts, arrivals, duration=1.0, purpose='train', cap=8):
    return {
        'schema_version': 1, 'model': ALIAS, 'max_model_len': 8192, 'duration_seconds': duration, 'output_cap': cap,
        'requests': [{'trace_id': f'r{index:02d}', 'arrival_seconds': arrival, 'input_tokens': len(prompt) + 4,
                      'max_tokens': cap, 'prompt': prompt}
                     for index, (prompt, arrival) in enumerate(zip(prompts, arrivals))],
        'provenance': {'purpose': purpose, 'tokenizer_sha256': FINGERPRINT},
    }


def make_binding(**changes):
    binding = {
        'schema': nc.BINDING_SCHEMA, 'served_alias': ALIAS, 'local_model_path': LOCAL, 'tokenizer_sha256': FINGERPRINT,
        'served_model_names': [ALIAS],
        'backend_argv': ['vllm', 'serve', LOCAL, '--served-model-name', ALIAS, '--scheduling-policy', 'fcfs',
                         '--no-enable-prefix-caching', '--max-model-len', '8192'],
        'backend_argv_source': 'literal CPU control (no launched backend)',
    }
    binding.update(changes)
    return binding


def make_comparability(ignore_eos=False, **model_changes):
    model = {'local_model_path': LOCAL, 'asset_sha256': '1' * 64, 'served_alias': ALIAS,
             'served_alias_verified': True, 'tokenizer_sha256': FINGERPRINT}
    return {'hardware': {'gpu': 'literal CPU control'}, 'model': dict(model, **model_changes),
            'backend': {'argv': 'literal'}, 'gateway': {'source_sha256': '2' * 64}, 'resources': {'cpu': 'literal'},
            'code': {'client': 'literal'}, 'client_generation': {'ignore_eos': ignore_eos,
                                                                 'max_tokens_policy': 'trace max_tokens'},
            'clock': 'literal monotonic'}


def chunk(content=None, finish=None, index=0, usage=None, choices=True):
    event = {'id': 'c', 'object': 'chat.completion.chunk', 'choices': [], 'usage': usage}
    if choices:
        delta = {} if content is None else {'content': content}
        event['choices'] = [{'index': index, 'delta': delta, 'finish_reason': finish}]
    return 'data: ' + json.dumps(event)


ROLE = 'data: ' + json.dumps({'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''},
                                           'finish_reason': None}]})
STOP = [ROLE, chunk('Hi'), chunk(' there', 'stop'), chunk(usage=USAGE, choices=False), DONE]
LENGTH = [chunk('a'), chunk('b', 'length'), chunk(usage=USAGE, choices=False), DONE]


def parse(lines):
    stream = nc.StreamParse()
    for at, line in enumerate(lines, start=1):
        stream.feed(line, float(at))
    return stream.result()


async def _body(lines, hang):
    for line in lines:
        yield (line + '\n\n').encode()
    if hang:
        await asyncio.sleep(30)


def sse(lines, hang=False):
    return httpx.Response(200, headers={'content-type': 'text/event-stream', 'x-request-id': 'gw-1'},
                          content=_body(lines, hang))


def refuse_connect(request):
    raise httpx.ConnectError('literal connection refused', request=request)


def interrupt(request):
    os.kill(os.getpid(), signal.SIGINT)
    return sse([chunk('a')], hang=True)


# Literal shape of the root gateway.py (0b1a80ad lines 653-692) admission 503 body; values are CPU literals.
GATEWAY_REJECTION = {'detail': nc.ADMISSION_DETAIL, 'estimated_request_cost': 1.5, 'estimated_input_tokens': 9,
                     'estimated_output_tokens': 8, 'max_admitted_cost': 16.0, 'budget_version': 3}
REJECTION_TEXT = json.dumps(GATEWAY_REJECTION, separators=(',', ':'))


async def _failing_body():
    yield b'{"detail":'
    raise httpx.ReadError('literal body read failure')


BEHAVIOR = {
    'stop': lambda request: sse(STOP), 'length': lambda request: sse(LENGTH),
    'admission': lambda request: httpx.Response(503, headers={'x-request-id': 'gw-rejected'},
                                                content=REJECTION_TEXT.encode()),
    'busy': lambda request: httpx.Response(503, json={'detail': 'overloaded'}),
    'server': lambda request: httpx.Response(500, text='literal failure'),
    'bodyfail': lambda request: httpx.Response(503, content=_failing_body()),
    'connect': refuse_connect, 'nodone': lambda request: sse(STOP[:-1]),
    'hang': lambda request: sse([chunk('a')], hang=True), 'interrupt': interrupt,
}


def make_handler(seen, models=None):
    async def handler(request):
        if request.method == 'GET' and request.url.path == nc.MODELS_PATH:
            seen.append(('models', str(request.url)))
            return httpx.Response(200, json=MODELS if models is None else models)
        body = json.loads(request.content)
        seen.append((body, dict(request.headers), str(request.url)))
        return BEHAVIOR[body['messages'][0]['content']](request)
    return handler


def run_metrics(records, start, duration, arm='fixed16'):
    rows = [nc.normalized_row(record) for record in records]
    end = max([start + duration] + [row['terminal_at'] for row in rows if row['terminal_at'] is not None])
    control = {'kind': 'static_fixed', 'budget': 16.0, 'basis': 'literal'} if arm == 'fixed16' else None
    return nm.compute({
        'arm': arm, 'purpose': 'train', 'clock_domain': 'literal', 'offered_start': start,
        'offered_duration_s': duration, 'envelope_start': start, 'envelope_end': end,
        'envelope_scope': 'offered_cohort_observed', 'rows': rows, 'slo_missing': None, 'control': control,
        'control_missing': None if control else 'literal: no controller',
        'provenance': {'offered_trace_sha256': 'x', 'offered_count': len(rows),
                       'offered_request_ids_sha256': nm.request_ids_sha256(row['request_id'] for row in rows)},
    })


def test_stop_stream_evidence_ignores_non_data_lines_and_same_chunk_finish():
    result = parse([': keepalive', 'id: 7', 'event: chunk', 'retry: 5', ''] + STOP)
    assert result['anomalies'] == [] and result['stream_done'] is True and result['done_at'] == 10.0
    assert (result['finish_reason'], result['finish_reason_evidence'], result['finish_reasons_seen']) == (
        'stop', 'observed', ['stop'])
    assert (result['first_content_at'], result['last_content_at']) == (7.0, 8.0)
    assert (result['completion_tokens'], result['completion_tokens_evidence'], result['prompt_tokens']) == (
        2, 'usage', 9)
    assert (result['data_events'], result['content_events'], result['finish_reason_events']) == (4, 2, 1)


PARSE_CASES = {
    'length': (LENGTH, 'length', 'observed', [], 2, 'usage'),
    'finish_absent': ([chunk('a'), chunk(usage=USAGE, choices=False), DONE], None, 'absent', [], 2, 'usage'),
    'two_finish_values': ([chunk('a', 'stop'), chunk(None, 'length'), DONE], None, 'contradiction',
                          ['finish_reason_contradiction'], None, 'absent'),
    'content_after_finish': ([chunk('a', 'stop'), chunk('b'), DONE], None, 'contradiction',
                             ['finish_reason_contradiction'], None, 'absent'),
    'error_event': ([chunk('a'), 'data: ' + json.dumps({'error': {'message': 'boom'}}), DONE], None, 'absent',
                    ['error_event'], None, 'absent'),
    'invalid_json': ([chunk('a', 'stop'), 'data: {not json', DONE], 'stop', 'observed', ['invalid_sse'], None,
                     'absent'),
    'nan_constant': ([chunk('a', 'stop'), 'data: {"choices": [], "usage": {"completion_tokens": NaN}}', DONE],
                     'stop', 'observed', ['invalid_sse'], None, 'absent'),
    'null_choices': ([chunk('a'), 'data: {"choices": null}', DONE], None, 'absent', ['invalid_sse'], None, 'absent'),
    'missing_done': ([chunk('a', 'stop'), chunk(usage=USAGE, choices=False)], 'stop', 'observed', ['missing_done'],
                     2, 'usage'),
    'empty_content': ([ROLE, chunk(None, 'stop'), DONE], 'stop', 'observed', ['empty_content'], None, 'absent'),
    'data_after_done': ([chunk('a', 'stop'), DONE, chunk('b')], 'stop', 'observed', ['data_after_done'], None,
                        'absent'),
    'multiple_usage': ([chunk('a', 'stop'), chunk(usage=USAGE, choices=False), chunk(usage=USAGE, choices=False),
                        DONE], 'stop', 'observed', [], None, 'multiple_usage'),
    'bool_usage': ([chunk('a', 'stop'), chunk(usage={'completion_tokens': True}, choices=False), DONE], 'stop',
                   'observed', [], None, 'invalid_usage'),
    'choice_index_1': ([chunk('a', 'stop', index=1), DONE], 'stop', 'observed', ['unexpected_choice_index'], None,
                       'absent'),
    'precedence': ([chunk('a'), 'data: {bad'], None, 'absent', ['invalid_sse', 'missing_done'], None, 'absent'),
}


@pytest.mark.parametrize('name', sorted(PARSE_CASES))
def test_stream_parse_literal_edge_cases(name):
    lines, finish, evidence, anomalies, tokens, tokens_evidence = PARSE_CASES[name]
    result = parse(lines)
    assert (result['finish_reason'], result['finish_reason_evidence'], result['anomalies'],
            result['completion_tokens'], result['completion_tokens_evidence']) == (
        finish, evidence, anomalies, tokens, tokens_evidence)


def _replay(trace, seen, lead=0.05, drain=0.3, before=None):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(make_handler(seen))) as client:
            start = time.monotonic() + lead
            records = nc.prepare_records(trace, start, RUN_ID)
            helper = asyncio.create_task(before(start)) if before else None
            deadline = await nc.replay(client, trace, records, gateway_url=GATEWAY, offered_start=start,
                                       drain_timeout=drain)
            if helper:
                await helper
            return start, records, deadline
    return asyncio.run(scenario())


def test_replay_retains_every_fate_payload_and_absolute_schedule():
    prompts = ['stop', 'admission', 'busy', 'server', 'connect', 'length', 'nodone', 'hang']
    arrivals = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    trace = tw.validate_trace(make_trace(prompts, arrivals, duration=0.2))
    seen = []
    start, records, deadline = _replay(trace, seen)
    by_prompt = dict(zip(prompts, records))
    assert {prompt: (record['fate'], record['fate_reason']) for prompt, record in by_prompt.items()} == {
        'stop': ('complete', 'complete'), 'admission': ('rejected', 'admission_capacity_503'),
        'busy': ('http_error', 'http_503'), 'server': ('http_error', 'http_500'),
        'connect': ('transport_error', 'before_response_headers'), 'length': ('complete', 'complete'),
        'nodone': ('stream_error', 'missing_done'), 'hang': ('cancelled', 'response_body')}
    assert (by_prompt['stop']['finish_reason'], by_prompt['length']['finish_reason']) == ('stop', 'length')
    admission = by_prompt['admission']
    assert admission['rejection'] == dict(GATEWAY_REJECTION, fields_absent=[])
    assert (admission['request_id'], admission['body_text'], admission['body_text_null_reason'],
            admission['body_sha256']) == ('gw-rejected', REJECTION_TEXT, None,
                                          hashlib.sha256(REJECTION_TEXT.encode()).hexdigest())
    assert by_prompt['busy']['rejection'] is None and by_prompt['stop']['request_id'] == 'gw-1'
    assert by_prompt['hang']['first_content_at'] is not None and by_prompt['hang']['terminal_at'] >= deadline - 0.01
    assert by_prompt['stop']['prompt_tokens_match'] is False  # literal usage prompt_tokens 9 != trace input 8
    for arrival, record in zip(arrivals, records):
        assert record['scheduled_at'] == start + arrival
        assert record['sent_at'] >= record['scheduled_at'] and record['send_lag_s'] >= 0
        assert record['terminal_at'] >= record['sent_at']
    assert len(seen) == len(prompts)
    for body, headers, url in seen:
        prompt = body['messages'][0]['content']
        assert url == GATEWAY + nc.CHAT_PATH and headers[nc.REQUEST_ID_HEADER] == by_prompt[prompt]['client_request_id']
        assert body == {'model': ALIAS, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
                        'max_tokens': 8, 'stream': True, 'stream_options': {'include_usage': True},
                        'ignore_eos': False}
    metrics = run_metrics(records, start, 0.2)
    assert metrics['all_offered'] == 8 and metrics['outcomes']['fates'] == {
        'complete': 2, 'rejected': 1, 'http_error': 2, 'stream_error': 1, 'transport_error': 1, 'cancelled': 1,
        'not_launched': 0, 'unknown': 0}


def test_rejection_body_retention_is_verbatim_and_never_invented():
    partial = {'detail': nc.ADMISSION_DETAIL, 'estimated_request_cost': 2.0, 'budget_version': 7}
    assert nc._admission_rejection(503, json.dumps(partial).encode()) == dict(
        partial, estimated_input_tokens=None, estimated_output_tokens=None, max_admitted_cost=None,
        fields_absent=['estimated_input_tokens', 'estimated_output_tokens', 'max_admitted_cost'])
    assert nc._admission_rejection(500, REJECTION_TEXT.encode()) is None
    assert nc._admission_rejection(503, json.dumps(dict(GATEWAY_REJECTION, detail='overloaded')).encode()) is None
    big = b'x' * nc.MAX_RETAINED_BODY_BYTES
    assert nc._retained_body(big) == (big.decode(), None)
    assert nc._retained_body(big + b'x') == (None, 'over_65536_bytes')
    assert nc._retained_body(b'\xff literal') == (None, 'not_utf8')
    prompts = ['busy', 'server', 'bodyfail', 'stop']
    trace = tw.validate_trace(make_trace(prompts, [0.0, 0.01, 0.02, 0.03], duration=0.1))
    records = dict(zip(prompts, _replay(trace, [])[1]))
    busy, server, failed, stop = (records[prompt] for prompt in prompts)
    assert (busy['fate'], busy['rejection'], json.loads(busy['body_text'])) == (
        'http_error', None, {'detail': 'overloaded'})
    assert busy['body_sha256'] == hashlib.sha256(busy['body_text'].encode()).hexdigest()
    assert (server['body_text'], server['body_text_null_reason']) == ('literal failure', None)
    assert (failed['fate'], failed['fate_reason'], failed['http_status'], failed['body_text'],
            failed['body_text_null_reason'], failed['body_bytes']) == (
        'transport_error', 'response_body', 503, None, 'body_read_incomplete', None)
    assert (stop['body_text'], stop['body_text_null_reason']) == (None, 'not_applicable')


def test_blocked_loop_sends_late_with_recorded_lag_never_early_or_retimed():
    trace = tw.validate_trace(make_trace(['stop', 'stop'], [0.0, 0.05], duration=0.1))

    async def block(start):
        await asyncio.sleep(max(0.0, start - 0.001 - time.monotonic()))
        time.sleep(0.15)

    start, records, _ = _replay(trace, [], drain=0.5, before=block)
    assert [record['fate'] for record in records] == ['complete', 'complete']
    assert [record['scheduled_at'] for record in records] == [start, start + 0.05]
    assert records[0]['send_lag_s'] >= 0.14 and records[1]['send_lag_s'] >= 0.09
    assert all(record['sent_at'] >= record['scheduled_at'] for record in records)


def test_passed_offered_start_is_refused_with_zero_sends():
    trace = tw.validate_trace(make_trace(['stop'], [0.0], duration=0.1))
    seen = []
    with pytest.raises(nc.ClientRefusal, match='already passed'):
        _replay(trace, seen, lead=-0.001)
    assert seen == []


def test_external_cancel_keeps_unsent_request_not_launched():
    trace = tw.validate_trace(make_trace(['stop', 'stop'], [0.0, 0.6], duration=1.0))

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(make_handler([]))) as client:
            start = time.monotonic() + 0.05
            records = nc.prepare_records(trace, start, RUN_ID)
            task = asyncio.create_task(nc.replay(client, trace, records, gateway_url=GATEWAY, offered_start=start,
                                                 drain_timeout=0.5))
            await asyncio.sleep(0.3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return start, records

    start, records = asyncio.run(scenario())
    assert records[0]['fate'] == 'complete'
    assert (records[1]['fate'], records[1]['fate_reason'], records[1]['sent_at']) == (
        'not_launched', 'cancelled_before_send', None)
    assert run_metrics(records, start, 1.0)['outcomes']['fates']['not_launched'] == 1


def test_literal_records_map_to_unchanged_natural_slo_with_all_offered_denominator():
    trace = make_trace(['a' * 10, 'b' * 600, 'c' * 3000], [0.0, 0.1, 0.2], duration=1.0, cap=128)
    records = nc.prepare_records(trace, 100.0, RUN_ID)
    for record, lines, sent, terminal in (
            (records[0], [(100.2, chunk('Hi')), (100.22, chunk('!', 'stop')),
                          (100.23, chunk(usage=USAGE, choices=False)), (100.235, DONE)], 100.01, 100.24),
            (records[1], [(100.4, chunk('a')), (100.45, chunk('b', 'length')),
                          (100.46, chunk(usage={'completion_tokens': 3}, choices=False)), (100.465, DONE)],
             100.11, 100.47)):
        stream = nc.StreamParse()
        for at, line in lines:
            stream.feed(line, at)
        nc._apply(record, stream.result())
        record.update(sent_at=sent, send_lag_s=sent - record['scheduled_at'], terminal_at=terminal,
                      fate='complete', fate_reason='complete')
    rows = [nc.normalized_row(record) for record in records]
    capture = {
        'schema': nc.SCHEMA, 'status': 'collected', 'arm': 'mpc', 'purpose': 'train', 'records': records,
        'rows': rows, 'clock_domain': 'literal', 'offered_start': 100.0, 'offered_duration_s': 1.0,
        'collection_end': 101.2, 'trace_sha256': 'f' * 64, 'trace_provenance': {}, 'run_id': RUN_ID,
        'binding_evidence': None, 'sources_sha256': {}, 'summary': nc.summarize_records(records),
        'offered_count': 3,
        'offered_request_ids_sha256': nm.request_ids_sha256(request['trace_id'] for request in trace['requests']),
    }
    for change, match in (({'offered_count': 4}, 'all-offered cohort: 3 rows'),
                          ({'offered_request_ids_sha256': nm.request_ids_sha256(['r00', 'r02', 'r01'])},
                           'request ids')):
        with pytest.raises(nc.ClientRefusal, match=match):
            nc.build_run(dict(capture, **change), control_missing='literal: no controller journal')
    with pytest.raises(nc.ClientRefusal, match='comparability contradicts'):
        nc.build_run(capture, control_missing='literal', comparability=make_comparability())
    assert capture['summary']['finish_reasons_of_complete'] == {'stop': 1, 'length': 1}
    metrics = nm.compute(nc.build_run(capture, control_missing='literal: no controller journal'))
    assert metrics['natural_slo']['natural_slo_good'] == 1
    assert metrics['natural_slo']['slo_attainment_fraction'] == pytest.approx(1 / 3)
    assert metrics['outcomes']['fates']['not_launched'] == 1 and metrics['windows']['envelope_end'] == 101.0
    assert metrics['ttft_s']['all_offered_penalized_scheduled']['n'] == 3
    assert metrics['controller_overhead']['null_reason'] == 'literal: no controller journal'
    assert {name: block['n_offered'] for name, block in metrics['groups'].items()} == {
        'le512': 1, '513to2048': 1, 'gt2048': 1}


def test_binding_accepts_public_alias_bound_to_local_assets():
    trace = make_trace(['stop'], [0.0])
    evidence = nc.check_binding(trace, make_binding())
    assert (evidence['served_alias'], evidence['local_model_path'], evidence['argv_scheduling_policy'],
            evidence['argv_prefix_caching_flags'], evidence['argv_max_model_len']) == (
        ALIAS, LOCAL, 'fcfs', ['--no-enable-prefix-caching'], '8192')
    both = make_binding(served_model_names=[ALIAS, LOCAL], backend_argv=[
        'python', '-m', 'vllm.entrypoints.openai.api_server', '--model', LOCAL, '--served-model-name', ALIAS, LOCAL])
    assert nc.check_binding(trace, both)['argv_scheduling_policy'] is None
    assert trace['model'] == ALIAS


BINDING_REFUSALS = {
    'unbound': (None, 'unbound served alias'),
    'extra_key': (dict(make_binding(), extra=1), 'binding keys'),
    'alias_differs': (make_binding(served_alias='other/model', served_model_names=['other/model']), 'trace.model'),
    'relative_path': (make_binding(local_model_path='models/Qwen3-4B-Instruct-2507'), 'absolute normalized'),
    'unnormalized_path': (make_binding(local_model_path='/workspace/../workspace/m'), 'absolute normalized'),
    'tokenizer_sha': (make_binding(tokenizer_sha256='0' * 64), 'tokenizer_sha256'),
    'alias_not_served': (make_binding(served_model_names=[LOCAL]), 'served_model_names'),
    'duplicate_names': (make_binding(served_model_names=[ALIAS, ALIAS]), 'served_model_names'),
    'argv_other_model': (make_binding(backend_argv=['vllm', 'serve', '/other', '--served-model-name', ALIAS]),
                         'does not launch'),
    'argv_equals_form': (make_binding(backend_argv=['vllm', 'serve', LOCAL, f'--served-model-name={ALIAS}']),
                         'exactly one'),
    'argv_repeated': (make_binding(backend_argv=['vllm', 'serve', LOCAL, '--served-model-name', ALIAS,
                                                 '--served_model_name', ALIAS]), 'exactly one'),
    'argv_names_differ': (make_binding(backend_argv=['vllm', 'serve', LOCAL, '--served-model-name', ALIAS, 'x']),
                          'exactly one'),
    'argv_source_empty': (make_binding(backend_argv_source=''), 'backend_argv_source'),
}


@pytest.mark.parametrize('name', sorted(BINDING_REFUSALS))
def test_binding_refusals(name):
    binding, match = BINDING_REFUSALS[name]
    with pytest.raises(nc.ClientRefusal, match=match):
        nc.check_binding(make_trace(['stop'], [0.0]), binding)


def test_served_models_and_tokenizer_refusals():
    binding = make_binding()
    assert nc.check_served_models(binding, MODELS, 8192)['alias_root'] == LOCAL
    entry = MODELS['data'][0]
    for models, match in (
            ({'data': [dict(entry, id='other')]}, 'ids differ'),
            ({'data': [entry, dict(entry, id='extra')]}, 'ids differ'),
            ({'data': [dict(entry, root=ALIAS)]}, 'root differs'),
            ({'data': [{key: value for key, value in entry.items() if key != 'max_model_len'}]}, 'max_model_len'),
            ([], 'not a model list')):
        with pytest.raises(nc.ClientRefusal, match=match):
            nc.check_served_models(binding, models, 8192)
    trace = make_trace(['abc'], [0.0])
    assert nc.check_tokenizer(trace, binding, FakeTokenizer())['prompts_verified'] == 1

    class OtherTemplate(FakeTokenizer):
        chat_template = 'other-template'

    with pytest.raises(nc.ClientRefusal, match='fingerprint'):
        nc.check_tokenizer(trace, binding, OtherTemplate())
    trace['requests'][0]['input_tokens'] += 1
    with pytest.raises(nc.ClientRefusal, match='token counts'):
        nc.check_tokenizer(trace, binding, FakeTokenizer())


def write_json(path, value):
    path.write_text(json.dumps(value))
    return str(path)


def cli_replay(tmp_path, name, prompts, arrivals, *, purpose='train', declared=None, arm='fixed16', binding=True,
               models=None, extra=()):
    trace_path = write_json(tmp_path / f'{name}.trace.json', make_trace(prompts, arrivals, 0.4, purpose))
    binding_path = tmp_path / f'{name}.binding.json'
    if binding:
        write_json(binding_path, make_binding())
    out, seen = tmp_path / name, []
    code = nc.main(['replay', '--trace', trace_path, '--binding', str(binding_path), '--gateway-url', GATEWAY,
                    '--backend-url', BACKEND, '--arm', arm, '--purpose', declared or purpose, '--output-dir',
                    str(out), '--start-delay', '0.3', '--drain-timeout', '0.5', *extra],
                   transport=httpx.MockTransport(make_handler(seen, models)),
                   tokenizer_loader=lambda path: FakeTokenizer() if path == LOCAL else None)
    capture_path = out / 'capture.json'
    return code, json.loads(capture_path.read_text()) if capture_path.exists() else None, seen, capture_path


def test_cli_replay_then_finalize_fixed_arm_only_with_driver_evidence(tmp_path, capsys):
    code, capture, seen, capture_path = cli_replay(tmp_path, 'fixed', ['stop', 'admission', 'length'],
                                                   [0.0, 0.05, 0.1])
    assert (code, capture['status'], capture['error']) == (0, 'collected', None)
    assert [record['fate'] for record in capture['records']] == ['complete', 'rejected', 'complete']
    assert capture['rows'] == [nc.normalized_row(record) for record in capture['records']]
    assert seen[0] == ('models', BACKEND + nc.MODELS_PATH) and len(seen) == 4
    assert capture['binding_evidence']['served_models']['alias_root'] == LOCAL
    assert capture['binding_evidence']['tokenizer']['prompts_verified'] == 3
    assert capture['payload_template']['model'] == ALIAS and set(capture['sources_sha256']) == set(nc.SOURCES)
    domain, start, end = capture['clock_domain'], capture['offered_start'], capture['collection_end']
    control = write_json(tmp_path / 'control.json', {'kind': 'static_fixed', 'budget': 16.0,
                                                     'basis': 'literal startup MAX_ADMITTED_COST'})
    evidence = {'acknowledged_at': start - 0.1, 'held_verified_at': end + 0.1, 'clock_domain': domain,
                'source': 'literal CPU control'}
    wall = {'start': start - 1.0, 'end': end + 0.2, 'clock_domain': domain, 'source': 'literal CPU control'}
    good = ['--control-json', control, '--control-evidence-json', write_json(tmp_path / 'evidence.json', evidence)]

    def bad_evidence(index, **change):
        return ['--control-json', control, '--control-evidence-json',
                write_json(tmp_path / f'evidence{index}.json', dict(evidence, **change))]

    capsys.readouterr()
    comparability = make_comparability()
    for index, (extra, match) in enumerate((
            (['--control-missing', 'none'], 'fixed arms require'),
            (bad_evidence(1, acknowledged_at=start), 'static_fixed control needs'),
            (bad_evidence(2, held_verified_at=end - 0.01), 'static_fixed control needs'),
            (bad_evidence(3, clock_domain='other'), 'static_fixed control needs'),
            (good + ['--full-wall-json', write_json(tmp_path / 'wall1.json', dict(wall, end=end - 0.01))],
             'full_wall must contain'),
            (good + ['--comparability-json', write_json(tmp_path / 'cmp1.json', make_comparability(
                local_model_path='/workspace/other'))], 'comparability contradicts'),
            (good + ['--comparability-json', write_json(tmp_path / 'cmp2.json', make_comparability(True))],
             'comparability contradicts'))):
        out = tmp_path / f'refused{index}'
        assert nc.main(['finalize', '--capture', str(capture_path), '--output-dir', str(out), *extra]) == 2
        assert match in capsys.readouterr().err and not out.exists()
    final = ['finalize', '--capture', str(capture_path), '--output-dir', str(tmp_path / 'final'), *good,
             '--full-wall-json', write_json(tmp_path / 'wall.json', wall),
             '--comparability-json', write_json(tmp_path / 'comparability.json', comparability)]
    assert nc.main(final) == 0
    run_bytes = (tmp_path / 'final' / 'run.json').read_bytes()
    run, metrics = json.loads(run_bytes), json.loads((tmp_path / 'final' / 'metrics.json').read_text())
    assert run['envelope_scope'] == 'full_wall' and run['provenance']['control_evidence'] == evidence
    assert (run['provenance']['comparability'], metrics['comparability']['status']) == (comparability, 'DECLARED')
    assert metrics['offered_cohort']['request_ids_sha256'] == nm.request_ids_sha256(['r00', 'r01', 'r02'])
    assert metrics['all_offered'] == 3 and metrics['outcomes']['fates']['rejected'] == 1
    assert metrics['windows']['full_wall_s'] == pytest.approx(wall['end'] - wall['start'])
    assert nc.main(final) == 2 and (tmp_path / 'final' / 'run.json').read_bytes() == run_bytes


def test_cli_refusals_send_nothing_and_preserve_capture(tmp_path):
    code, capture, seen, _ = cli_replay(tmp_path, 'unbound', ['stop'], [0.0], binding=False)
    assert (code, capture['status'], seen, capture['records']) == (2, 'refused', [], None)
    assert 'unreadable' in capture['error']
    code, capture, seen, _ = cli_replay(tmp_path, 'hfroot', ['stop'], [0.0],
                                        models={'data': [dict(MODELS['data'][0], root=ALIAS)]})
    assert (code, capture['status'], [item[0] for item in seen], capture['records']) == (2, 'refused', ['models'], None)
    assert 'root differs' in capture['error']
    code, capture, seen, _ = cli_replay(tmp_path, 'mislabel', ['stop'], [0.0], purpose='engineering', declared='train')
    assert (code, capture['status'], seen) == (2, 'refused', []) and 'provenance purpose' in capture['error']
    (tmp_path / 'existing').mkdir()
    (tmp_path / 'existing' / 'marker').write_text('kept')
    assert cli_replay(tmp_path, 'existing', ['stop'], [0.0])[:3] == (2, None, [])
    assert (tmp_path / 'existing' / 'marker').read_text() == 'kept'
    for flag, value in (('--purpose', 'heldout'), ('--purpose', 'eval_probe'), ('--purpose', 'Train'),
                        ('--drain-timeout', 'nan'), ('--start-delay', 'inf')):
        with pytest.raises(SystemExit) as exit_info:
            cli_replay(tmp_path, f'argparse-{value}', ['stop'], [0.0], extra=(flag, value))
        assert exit_info.value.code == 2 and not (tmp_path / f'argparse-{value}').exists()


def test_raw_engineering_purpose_is_captured_but_never_finalized(tmp_path, capsys):
    code, capture, _, capture_path = cli_replay(tmp_path, 'probe', ['stop'], [0.0], purpose='engineering',
                                                arm='mpc')
    assert (code, capture['status'], capture['purpose'], capture['records'][0]['fate']) == (
        0, 'collected', 'engineering', 'complete')
    capsys.readouterr()
    out = tmp_path / 'probe-final'
    assert nc.main(['finalize', '--capture', str(capture_path), '--output-dir', str(out),
                    '--control-missing', 'raw probe']) == 2
    assert 'raw engineering capture' in capsys.readouterr().err and not out.exists()


def test_sigint_interrupt_preserves_every_offered_record(tmp_path):
    before = signal.getsignal(signal.SIGINT)
    code, capture, _, _ = cli_replay(tmp_path, 'interrupted', ['interrupt', 'stop'], [0.0, 0.35])
    assert signal.getsignal(signal.SIGINT) is before
    assert (code, capture['status']) == (3, 'interrupted') and capture['error'].startswith('CancelledError')
    assert [(record['fate'], record['fate_reason']) for record in capture['records']] == [
        ('cancelled', 'response_body'), ('not_launched', 'cancelled_before_send')]
    assert capture['records'][0]['first_content_at'] is not None and capture['summary']['offered'] == 2
    with pytest.raises(nc.ClientRefusal, match='preserved as failure evidence'):
        nc.build_run(capture, control_missing='interrupted')


def lib_prepare(tmp_path, name, prompts, arrivals, duration=0.4, *, purpose='train', **binding_changes):
    trace_path = write_json(tmp_path / f'{name}.trace.json', make_trace(prompts, arrivals, duration, purpose))
    binding_path = write_json(tmp_path / f'{name}.binding.json', make_binding(**binding_changes))
    return nc.prepare_capture(trace_path=trace_path, binding_path=binding_path, arm='mpc', purpose=purpose,
                              gateway_url=GATEWAY, backend_url=BACKEND,
                              tokenizer_loader=lambda path: FakeTokenizer() if path == LOCAL else None)


def lib_replay(prepared, transport, **change):
    """capture_replay awaited in a fresh loop; change values may be callables of the planned offered start."""
    async def scenario():
        start = time.monotonic() + 0.2
        kwargs = dict(offered_start=start, deadline=start + 1.0, cleanup_reserve_s=0.1, clock_domain=nc.clock_domain())
        kwargs.update({key: value(start) if callable(value) else value for key, value in change.items()})
        return await nc.capture_replay(prepared, transport=transport, **kwargs)
    return asyncio.run(scenario())


def test_library_capture_inside_running_loop_keeps_caller_clock_and_cli_envelope(tmp_path):
    prepared = lib_prepare(tmp_path, 'lib', ['stop', 'admission', 'length', 'hang'], [0.0, 0.05, 0.1, 0.15])
    capture, seen, beats = prepared.capture, [], []
    assert (prepared.ready, capture['status'], capture['error'], capture['records']) == (
        True, 'failed', nc.NOT_RUN_ERROR, None)
    assert capture['binding_evidence']['tokenizer']['prompts_verified'] == 4

    def handlers():
        return signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)

    async def scenario():
        inside = handlers()

        async def heartbeat():
            while True:
                beats.append((time.monotonic(), handlers()))
                await asyncio.sleep(0.02)

        beat = asyncio.create_task(heartbeat())
        start = time.monotonic() + 0.3
        result = await nc.capture_replay(prepared, offered_start=start, deadline=start + 1.2, cleanup_reserve_s=0.5,
                                         clock_domain=nc.clock_domain(),
                                         transport=httpx.MockTransport(make_handler(seen)))
        left = [task for task in asyncio.all_tasks() if task not in (beat, asyncio.current_task())]
        beat.cancel()
        await asyncio.gather(beat, return_exceptions=True)
        return start, inside, result, left

    start, inside, result, left = asyncio.run(scenario())
    records = capture['records']
    assert result is capture and (capture['status'], capture['error'], left) == ('collected', None, [])
    assert (capture['offered_start'], capture['offered_end'], capture['drain_deadline']) == (
        start, start + 0.4, start + 1.2 - 0.5)
    assert {key: capture['config'][key] for key in ('entry', 'driver_offered_start', 'caller_deadline',
                                                    'cleanup_reserve_s', 'caller_clock_domain')} == {
        'entry': 'library', 'driver_offered_start': start, 'caller_deadline': start + 1.2, 'cleanup_reserve_s': 0.5,
        'caller_clock_domain': capture['clock_domain']}
    assert [(record['fate'], record['fate_reason']) for record in records] == [
        ('complete', 'complete'), ('rejected', 'admission_capacity_503'), ('complete', 'complete'),
        ('cancelled', 'response_body')]
    assert [record['scheduled_at'] for record in records] == [start + arrival for arrival in (0.0, 0.05, 0.1, 0.15)]
    # the hung request is cancelled at deadline - reserve, leaving the caller's cleanup reserve unused
    assert capture['drain_deadline'] - 0.01 <= records[3]['terminal_at'] < start + 1.2
    assert capture['rows'] == [nc.normalized_row(record) for record in records]
    assert capture['summary'] == nc.summarize_records(records)
    assert seen[0] == ('models', BACKEND + nc.MODELS_PATH) and len(seen) == 5
    assert {sample for _, sample in beats} == {inside}  # no signal handler installed or removed
    assert beats[0][0] < start and beats[-1][0] > capture['offered_end']  # the caller's own task kept running
    digest = nc.write_capture(tmp_path / 'lib.capture.json', capture)
    data = (tmp_path / 'lib.capture.json').read_bytes()
    assert digest == hashlib.sha256(data).hexdigest() and json.loads(data) == capture
    with pytest.raises(FileExistsError):
        nc.write_capture(tmp_path / 'lib.capture.json', capture)
    metrics = nm.compute(nc.build_run(json.loads(data), control_missing='literal: no controller journal'))
    assert metrics['all_offered'] == 4 and metrics['outcomes']['fates']['cancelled'] == 1
    cli = cli_replay(tmp_path, 'cli', ['stop'], [0.0], arm='mpc')[1]
    assert set(cli) == set(capture) and set(cli['config']) < set(capture['config'])
    frozen, sent = json.dumps(capture, sort_keys=True), []
    for name, change, match in (('truncated', {'deadline': lambda planned: planned + 0.45}, 'never truncated'),
                                ('passed', {'offered_start': lambda planned: planned - 0.5}, 'no rescheduling'),
                                ('domain', {'clock_domain': 'literal other host'}, 'clock_domain differs'),
                                ('nan', {'cleanup_reserve_s': float('nan')}, 'finite'),
                                ('negative', {'cleanup_reserve_s': -0.1}, 'finite')):
        refused = lib_replay(lib_prepare(tmp_path, name, ['stop'], [0.0]), httpx.MockTransport(make_handler(sent)),
                             **change)
        assert (refused['status'], refused['records'], sent) == ('refused', None, []) and match in refused['error']
        nc.write_capture(tmp_path / f'{name}.capture.json', refused)
    unbound = lib_prepare(tmp_path, 'unbound', ['stop'], [0.0], served_alias='other/alias')
    assert (unbound.ready, unbound.capture['status']) == (False, 'refused')
    assert 'served_alias' in unbound.capture['error']
    assert lib_prepare(tmp_path, 'heldout', ['stop'], [0.0], purpose='heldout').capture['status'] == 'refused'
    for unusable in (unbound, prepared):
        with pytest.raises(nc.ClientRefusal, match='ready, unused'):
            lib_replay(unusable, httpx.MockTransport(make_handler(sent)))
    assert sent == [] and json.dumps(capture, sort_keys=True) == frozen


class CloseFails(httpx.MockTransport):
    async def aclose(self):
        raise RuntimeError('literal transport close failure')


def test_library_cancellation_and_failure_keep_every_allocated_offer_after_joining_requests(tmp_path):
    prepared = lib_prepare(tmp_path, 'cancel', ['stop', 'hang', 'stop'], [0.0, 0.05, 0.6], duration=1.0)
    seen = []

    async def scenario():
        start = time.monotonic() + 0.2
        task = asyncio.create_task(nc.capture_replay(
            prepared, offered_start=start, deadline=start + 3.0, cleanup_reserve_s=0.5,
            clock_domain=nc.clock_domain(), transport=httpx.MockTransport(make_handler(seen))))
        await asyncio.sleep(max(0.0, start + 0.3 - time.monotonic()))
        for _ in range(3):  # repeated caller cancellation while the request tasks are being joined
            task.cancel()
            await asyncio.sleep(0)
        with pytest.raises(asyncio.CancelledError):
            await task
        left = [other for other in asyncio.all_tasks() if other is not asyncio.current_task()]
        frozen = json.dumps(prepared.capture, sort_keys=True)
        await asyncio.sleep(max(0.0, start + 0.75 - time.monotonic()))  # past r02's scheduled send
        return left, frozen

    left, frozen = asyncio.run(scenario())
    capture = prepared.capture
    assert (left, json.dumps(capture, sort_keys=True), len(seen)) == ([], frozen, 3)
    assert (capture['status'], capture['error']) == ('interrupted', 'CancelledError: replay interrupted')
    assert [(record['fate'], record['fate_reason']) for record in capture['records']] == [
        ('complete', 'complete'), ('cancelled', 'response_body'), ('not_launched', 'cancelled_before_send')]
    assert capture['rows'] == [nc.normalized_row(record) for record in capture['records']]
    assert (capture['summary']['offered'], capture['summary']['sent']) == (3, 2)
    with pytest.raises(nc.ClientRefusal, match='preserved as failure evidence'):
        nc.build_run(capture, control_missing='interrupted')
    failed = lib_replay(lib_prepare(tmp_path, 'closefail', ['stop'], [0.0]), CloseFails(make_handler([])))
    assert (failed['status'], failed['error']) == ('failed', 'RuntimeError: literal transport close failure')
    assert [record['fate'] for record in failed['records']] == ['complete'] and failed['summary']['offered'] == 1
    assert failed['rows'] == [nc.normalized_row(record) for record in failed['records']]
